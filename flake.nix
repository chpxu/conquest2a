{
  description = "CONQUEST2a";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/";
    nixpkgs2505.url = "github:NixOS/nixpkgs/2b0d2b456e4e8452cf1c16d00118d145f31160f9"; # to use for older packages
    flake-parts.url = "github:hercules-ci/flake-parts/a34fae9c08a15ad73f295041fec82323541400a9";
    devshell.url = "github:numtide/devshell/17ed8d9744ebe70424659b0ef74ad6d41fc87071";
    import-tree.url = "github:vic/import-tree/3c23749d8013ec6daa1d7255057590e9ca726646";
    git-hooks-nix.url = "github:cachix/git-hooks.nix/b68b780b69702a090c8bb1b973bab13756cc7a27";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };
  outputs =
    {
      nixpkgs2505,
      flake-parts,
      ...
    }@inputs:
    let
      inherit (inputs) import-tree;
      getLanguageDefaultNix = (import-tree.match ".*/default\\.nix") ./nix/languages;
      getEditorDefaultNix = (import-tree.match ".*/default\\.nix") ./nix/editors;
      imports = builtins.concatLists [
        [
          inputs.flake-parts.flakeModules.easyOverlay
          inputs.devshell.flakeModule
          inputs.treefmt-nix.flakeModule
          inputs.git-hooks-nix.flakeModule
        ]
        getLanguageDefaultNix.imports
        getEditorDefaultNix.imports
      ];
    in
    flake-parts.lib.mkFlake
      {
        inherit inputs;
      }
      {
        inherit imports;
        systems = [
          "x86_64-linux"
        ];
        perSystem =
          {
            config,
            inputs',
            self',
            pkgs,
            system,
            ...
          }:
          let
            userConfig = import ./config.nix { inherit pkgs config; };
          in
          {
            _module.args = {
              pkgsOlder = import nixpkgs2505 {
                inherit system inputs';
              };
              helper = import ./nix/helpers;
              pkgs = import inputs.nixpkgs {
                inherit system;
                overlays = [
                  (_final: prev: {
                    black = prev.black.overrideAttrs {
                      src = pkgs.fetchFromGitHub {
                        owner = "psf";
                        repo = "black";
                        tag = "26.5.1";
                        sha256 = "sha256-xALg9ta0U2V6i/b7VYiPKu0oNnHfg9T+XuK3CvqJmjs=";
                      };
                    };
                  })
                ];
                config = { };
              };
            };
            imports = [ userConfig ]; # settings from config.nix defined by user
            packages.default = pkgs."python${config.languages.python.version}Packages".buildPythonPackage {
              pname = "conquest2a";
              version = "0.4.0";
              pyproject = true;
              src = ./.;
              buildInputs =
                with pkgs."python${config.languages.python.version}Packages";
                [
                  numpy
                  scipy
                  hatchling
                  ase
                  matplotlib
                ]
                ++ [ self'.packages.scienceplots ];
              enableParallelBuilding = true;
            };
            packages.scienceplots = pkgs."python${config.languages.python.version}Packages".buildPythonPackage {
              pname = "SciencePlots";
              version = "2.2.2";
              pyproject = true;
              src = pkgs.fetchFromGitHub {
                owner = "garrettj403";
                repo = "SciencePlots";
                rev = "b9b16959570bd2fbc9ff5118bacc423c3bddd592";
                sha256 = "sha256-Sj0SdTu0M0wgTiUuC9ad73W8olsnbjzJgkaIsYKPYvo=";
              };
              build-system = with pkgs."python${config.languages.python.version}Packages"; [
                setuptools
                setuptools-scm
              ];
              dependencies = with pkgs."python${config.languages.python.version}Packages"; [
                matplotlib
                setuptools-scm
              ];
              pythonImportsCheck = [ "scienceplots" ];
              doCheck = false; # no tests
            };
            pre-commit.settings.hooks = {
              nixfmt.enable = true;
              flake-checker = {
                enable = true;
              };
              mypy.enable = true;
              flake8.enable = true;
              pylint.enable = true;
              treefmt = {
                enable = true;
                package = config.treefmt.build.wrapper;
              };
            };
            treefmt = {
              projectRootFile = "flake.nix";
              programs = {
                deadnix.enable = true;
                statix.enable = true;
                nixfmt.enable = true;
                black.enable = true;
              };

              settings = {
                global.excludes = [
                  ".direnv/*"
                  "examples/*"
                ];

                formatter = {
                  deadnix.priority = 1;
                  statix.priority = 2;
                  nixfmt = {
                    priority = 3;
                    strict = true;
                    indent = 2;
                  };
                  black = {
                    priority = 4;
                  };
                };
              };
            };
          };
        flake = {
          templates = {
            default = {
              description = ''
                Opinionated flake
              '';
              path = ./.;
              welcomeText = ''
                Welcome to devflake. Edit flake.nix to get started. See the README.md for more information.
              '';
            };
          };
        };
      };
}
